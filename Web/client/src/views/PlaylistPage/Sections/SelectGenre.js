import React, {useState} from "react";
import {Alert, Button, CardBody, CardHeader, Table} from "reactstrap";

import Polarizing from "components/Polarizing.js";


const SelectGenre = (props) => {

    const genres = ["발라드", "댄스", "랩/힙합", "R&B/Soul", "인디음악", "록/메탈", "트로트", "포크/블루스"];

    const [ButtonAction, setButtonAction] = useState(false);

    const [LikeGenre, setLikeGenre] = useState(new Set());
    const [DislikeGenre, setDislikeGenre] = useState(new Set());

    const addGenre = (genre, value) => {

        deleteGenre(genre);

        if (value === 'like') {

            LikeGenre.add(genre);
            setLikeGenre(LikeGenre);
        }

        if (value === 'dislike') {

            DislikeGenre.add(genre);
            setDislikeGenre(DislikeGenre);
        }

        if (LikeGenre.size >= 3 && DislikeGenre.size >= 3) {
            onChangeBtnState(true);
        } else {
            onChangeBtnState(false);
        }
    };

    const deleteGenre = (genre) => {

        if (LikeGenre.has(genre)) {

            LikeGenre.delete(genre);
            setLikeGenre(LikeGenre);
        }

        if (DislikeGenre.has(genre)) {

            DislikeGenre.delete(genre);
            setDislikeGenre(DislikeGenre);
        }
    };

    const onChangeBtnState = (state) => {

        setButtonAction(state);
    };

    const onNextPage = () => {

        props.onChangePage([...LikeGenre], [...DislikeGenre]);
    };

    const renderGenre = genres && genres.map((g, index) => {
        return (
            <tr key={index}>
                <td>
                    {g}
                </td>
                <td className="text-right">
                    <Polarizing id={g} addList={addGenre}/>
                </td>
            </tr>
        );
    });

    return (
        <>
            <CardHeader className="border-0">
                <Alert color="primary">
                    선호하는 장르와 선호하지 않는 장르를 선택해 주세요. (각 3개 이상)
                </Alert>
            </CardHeader>
            <Table className="align-items-center table-flush" responsive>
                <thead className="thead-light">
                <tr>
                    <th scope="col">장르</th>
                    <th scope="col"/>
                </tr>
                </thead>
                {
                    renderGenre ? (<tbody>{renderGenre}</tbody>) : null
                }
            </Table>
            <CardBody>
                <div className="text-right">
                    <Button color="info" disabled={!ButtonAction} type="button" onClick={onNextPage}>
                        다음
                    </Button>
                </div>
            </CardBody>
        </>
    );
}

export default SelectGenre;
