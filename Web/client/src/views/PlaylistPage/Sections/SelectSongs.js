import React, {useEffect, useState} from "react";
import {Alert, Button, CardBody, CardHeader} from "reactstrap";

import SongInfo from "components/SongInfo.js";


const SelectGenre = (props) => {

    const [FinishedAction, setFinishedAction] = useState(false);

    const [SongList, setSongList] = useState([]);

    const [LikeSong, setLikeSong] = useState([]);
    const [DislikeSong, setDislikeSong] = useState([]);

    const loadJsonFile = (tag) => {

        switch (tag) {

            case "발라드":
                return require('../JsonFiles/발라드_GN0100.json');

            case "댄스":
                return require("../JsonFiles/댄스_GN0200.json");

            case "랩/힙합":
                return require("../JsonFiles/랩-힙합_GN0300_GN1200.json");

            case "R&B/Soul":
                return require("../JsonFiles/R&B-Soul_GN0400_GN1300.json");

            case "인디음악":
                return require("../JsonFiles/인디음악_GN0500.json");

            case "록/메탈":
                return require("../JsonFiles/록-메탈_GN0600_GN1000.json");

            case "트로트":
                return require("../JsonFiles/트로트_GN0700.json");

            case "포크/블루스":
                return require("../JsonFiles/포크-블루스_GN0800_GN1400.json");

            default:
                return null;
        }
    };

    // eslint-disable-next-line react-hooks/exhaustive-deps
    const makeSongList = (tag) => {

        const json = loadJsonFile(tag);

        return Object.keys(json["id"]).map((key, _) => {

            let song = {};

            song["song_id"] = key;
            song["song_name"] = json["song_name"][key];
            song["artist"] = json["artist_name_basket"][key];
            song["album_name"] = json["album_name"][key];

            return song;
        });
    };

    useEffect(() => {

        let list = [];

        for (const tag of props.tags) {

            list = [...list, ...makeSongList(tag)];
        }
        setSongList(list);
    }, [props.tags]);

    const onChangeBtnState = (state) => {

        setFinishedAction(state);
    };

    const onSyncList = (likes, dislikes) => {

        setLikeSong(likes);
        setDislikeSong(dislikes);

        if (likes.length >= 5 && dislikes.length >= 5) {
            onChangeBtnState(true);
        } else {
            onChangeBtnState(false);
        }
    };

    const onSubmit = () => {

        props.onSubmit([...LikeSong], [...DislikeSong]);
    };

    return (
        <>
            <CardHeader className="border-0">
                <Alert className="alert-default">
                    다음 곡들에 대해 <strong>좋아요</strong> 또는 <strong>싫어요</strong>를 표시해 주세요. (각 5개 이상)
                </Alert>
            </CardHeader>
            {
                SongList ? (
                    <>
                        <SongInfo song_list={SongList} onSyncList={onSyncList}/>
                        <CardBody>
                            <div className="text-right">
                                <Button color="success" disabled={!FinishedAction} type="button" onClick={onSubmit}>
                                    완료
                                </Button>
                            </div>
                        </CardBody>
                    </>
                ) : null
            }
        </>
    );
}

export default SelectGenre;
