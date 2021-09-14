/*!

=========================================================
* Argon Dashboard React - v1.2.1
=========================================================

* Product Page: https://www.creative-tim.com/product/argon-dashboard-react
* Copyright 2021 Creative Tim (https://www.creative-tim.com)
* Licensed under MIT (https://github.com/creativetimofficial/argon-dashboard-react/blob/master/LICENSE.md)

* Coded by Creative Tim

=========================================================

* The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

*/

import React, {useEffect, useState} from "react";

// reactstrap components
import {Card, CardHeader, Col, Container, Row} from "reactstrap";

// core components
import UserHeader from "components/Headers/UserHeader.js";

import axios from "axios";

import SongInfo from "../../components/SongInfo";


const Profile = () => {

    const [LikeList, setLikeList] = useState(null);
    const [DislikeList, setDislikeList] = useState(null);

    useEffect(() => {

        axios.get('http://127.0.0.1:8000/users/')
            .then(response => {
                if (response.data) {
                    setLikeList(response.data.like);
                    setDislikeList(response.data.dislike);
                } else {
                    alert('Playlist(song_list)를 가져오지 못했습니다.');
                }
            })
            .catch(error => {
                console.log(error);
            })
    }, []);

    return (
        <>
            <UserHeader/>
            {/* Page content */}
            <Container className="mt--7" fluid>
                <Row>
                    <Col>
                        <Card className="shadow">
                            <CardHeader className="border-0">
                                <h3 className="mb-0">좋아하는 곡 목록</h3>
                            </CardHeader>
                            {
                                LikeList ? <SongInfo song_list={LikeList}/> : null
                            }
                        </Card>
                    </Col>
                    <Col>
                        <Card className="shadow">
                            <CardHeader className="border-0">
                                <h3 className="mb-0">좋아하지 않는 곡 목록</h3>
                            </CardHeader>
                            {
                                DislikeList ? <SongInfo song_list={DislikeList}/> : null
                            }
                        </Card>
                    </Col>
                </Row>
            </Container>
        </>
    );
};

export default Profile;
